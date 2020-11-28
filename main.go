package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gorilla/mux"
	"github.com/joho/godotenv"
)

func sendJSONResponse(w http.ResponseWriter, data interface{}) {
	body, err := json.Marshal(data)
	buff := bytes.NewBuffer(body)

	if err != nil {
		log.Printf("Failed to encode a JSON response: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=UTF-8")
	w.WriteHeader(http.StatusOK)
	_, err = w.Write(buff.Bytes())
	if err != nil {
		log.Printf("Failed to write the response body: %v", err)
		return
	}
}

func Inference(w http.ResponseWriter, req *http.Request) {
	var data InputData
	body, _ := ioutil.ReadAll(req.Body)
	json.Unmarshal(body, &data)

	result := naiveBayes(data)
	sendJSONResponse(w, Response{
		Class:       os.Getenv("CLASS"),
		Probability: result,
	})
	return
}

func HeartBeat(w http.ResponseWriter, req *http.Request) {
	sendJSONResponse(w, map[string]string{
		"message": "ok",
	})
	return
}

func makeRouter() *mux.Router {
	router := mux.NewRouter()
	_ = &http.Server{
		ReadTimeout:  300 * time.Second,
		WriteTimeout: 300 * time.Second,
	}

	//endpoints
	router.HandleFunc("/send", Inference).Methods("POST")
	router.HandleFunc("/heartbeat", HeartBeat).Methods("GET")
	return router
}

func main() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	fileName := fmt.Sprintf("./%s.csv", os.Getenv("CLASS"))
	if _, err := os.Stat(fileName); os.IsNotExist(err) {
		fmt.Println("Training...")
		start := time.Now()
		train(100)
		fmt.Printf("Took %v\n", time.Since(start))
		fmt.Println("Trained")
	}

	router := makeRouter()
	address := fmt.Sprintf(":%s", os.Getenv("ADDRESS"))
	if err = http.ListenAndServe(address, router); err != nil {
		fmt.Println(err)
	}
}
