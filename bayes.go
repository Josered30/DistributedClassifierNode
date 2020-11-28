package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/bbalet/stopwords"
)

type Response struct {
	Class       string  `json:"class"`
	Probability float64 `json:"probability"`
}

type InputData struct {
	Description string  `json:"description"`
	Headline    string  `json:"headline"`
	Rate        float64 `json:"rate"`
}

type Data struct {
	Description string `json:"short_description"`
	Headline    string `json:"headline"`
	Date        string `json:"date"`
	Link        string `json:"link"`
	Author      string `json:"authors"`
	Category    string `json:"category"`
}

func tokenize(message string) []string {
	cleanMessage := stopwords.CleanString(message, "en", false)

	var re = regexp.MustCompile(`\$[\d]`)
	cleanMessage = re.ReplaceAllString(cleanMessage, "price")

	re = regexp.MustCompile(`\%[\d]`)
	cleanMessage = re.ReplaceAllString(cleanMessage, "percentage")

	re = regexp.MustCompile(`http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+`)
	cleanMessage = re.ReplaceAllString(cleanMessage, "url")

	re = regexp.MustCompile(`www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+`)
	cleanMessage = re.ReplaceAllString(cleanMessage, "url")

	re = regexp.MustCompile(`(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)`)
	cleanMessage = re.ReplaceAllString(cleanMessage, "email")

	re = regexp.MustCompile(`\b\w{1,2}\b`)
	cleanMessage = re.ReplaceAllString(cleanMessage, " ")

	re = regexp.MustCompile(`[\W\d]`)
	cleanMessage = re.ReplaceAllString(cleanMessage, " ")

	re = regexp.MustCompile(`\s+`)
	cleanMessage = re.ReplaceAllString(cleanMessage, " ")

	cleanMessage = strings.TrimSpace(cleanMessage)
	parts := strings.Split(cleanMessage, " ")
	return parts
}

func lineCounter(fileName string) (int, error) {
	file, _ := os.Open(fileName)
	defer file.Close()

	r := bufio.NewReader(file)
	buf := make([]byte, 32*1024)
	count := 0
	lineSep := []byte{'\n'}

	for {
		c, err := r.Read(buf)
		count += bytes.Count(buf[:c], lineSep)

		switch {
		case err == io.EOF:
			return count, nil

		case err != nil:
			return count, err
		}
	}
}

func savefile(filename string, data map[string]float64) {
	csvFile, err := os.Create(filename)
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}
	defer csvFile.Close()

	csvwriter := csv.NewWriter(csvFile)
	defer csvwriter.Flush()

	csvwriter.Write([]string{"word", "weight"})

	for i, d := range data {
		_ = csvwriter.Write([]string{i, strconv.FormatFloat(d, 'f', 10, 64)})
	}
}

func dataParser(texts []string, wg *sync.WaitGroup, mutex *sync.Mutex, data *map[string]int, flag bool) {
	for _, text := range texts {
		tokens := tokenize(text)
		for _, token := range tokens {
			mutex.Lock()

			_, ok := (*data)[token]
			if ok && flag {
				(*data)[token]++
			} else {
				(*data)[token] = 0
			}
			mutex.Unlock()
		}
	}
	wg.Done()
}

func getVocabulary(lines, size int) map[string]int {
	file, _ := os.Open("data.json")
	scanner := bufio.NewScanner(file)
	defer file.Close()

	var wg sync.WaitGroup
	var mutex sync.Mutex
	var counter int
	var aux []string

	vocabulary := make(map[string]int)
	for scanner.Scan() {
		var data Data
		if err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
			//handle error
		} else {
			aux = append(aux, data.Description+" "+data.Headline)
			left := lines - counter
			if left < size || counter%size == 0 {
				wg.Add(1)
				go dataParser(aux, &wg, &mutex, &vocabulary, true)
				aux = nil
			}
		}
		counter++
	}

	if aux != nil {
		wg.Add(1)
		dataParser(aux, &wg, &mutex, &vocabulary, false)
	}

	wg.Wait()
	return vocabulary
}

func csvReader(filename string) ([][]string, error) {
	recordFile, err := os.Open(filename)
	defer recordFile.Close()

	if err != nil {
		fmt.Println("Error:", err)
		return nil, err
	}
	reader := csv.NewReader(recordFile)
	records, _ := reader.ReadAll()
	return records, err

}

func loadData(filename string) (map[string]float64, error) {
	records, err := csvReader(filename)
	if err != nil {
		return nil, err
	}
	records = records[1:]
	dataMap := make(map[string]float64)

	for _, record := range records {
		prob, _ := strconv.ParseFloat(record[1], 64)
		dataMap[record[0]] = prob
	}
	return dataMap, err
}

func train(chunks int) {
	lines, _ := lineCounter("data.json")
	size := lines / chunks

	vocabulary := getVocabulary(lines, size)
	class := os.Getenv("CLASS")

	file, _ := os.Open("data.json")
	scanner := bufio.NewScanner(file)
	defer file.Close()

	var wg sync.WaitGroup
	var mutex sync.Mutex
	var counter int
	var aux []string

	for scanner.Scan() {
		var data Data
		if err := json.Unmarshal(scanner.Bytes(), &data); err != nil {
			//handle error
		} else {
			if data.Category == class {
				aux = append(aux, data.Description+" "+data.Headline)
			}
			left := lines - counter
			if left < size || counter%size == 0 {
				wg.Add(1)
				go dataParser(aux, &wg, &mutex, &vocabulary, true)
				aux = nil
			}
		}
		counter++
	}

	wg.Wait()
	var wordCount int
	for _, value := range vocabulary {
		wordCount += value
	}

	probabilities := make(map[string]float64)
	for key, value := range vocabulary {
		probabilities[key] += (float64(value) + 1) / float64(wordCount+len(vocabulary))
	}

	fileName := fmt.Sprintf("%s.csv", class)
	savefile(fileName, probabilities)
}

func naiveBayes(inputData InputData) float64 {
	class := os.Getenv("class")
	probs, _ := loadData(fmt.Sprintf("./%s.csv", class))

	var aux string = inputData.Description + " " + inputData.Headline
	tokens := tokenize(aux)

	var result float64

	for _, token := range tokens {
		if val, ok := probs[token]; ok {
			result += val
		}
	}
	return result
}
