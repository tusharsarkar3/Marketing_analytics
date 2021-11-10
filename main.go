package main

import (
	"encoding/csv"
	"fmt"
	"html/template"
	"log"
	"net/http"
	"os"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/trees"
)

var tpl *template.Template

func init() {
	tpl = template.Must(template.ParseGlob("templates/*"))
}

func main() {
	http.HandleFunc("/results", results)
	http.HandleFunc("/", getdata)
	http.Handle("/favicon.ico", http.NotFoundHandler())
	http.ListenAndServe(":8080", nil)
	//train()
}

func getdata(w http.ResponseWriter, req *http.Request) {
	//train()
	tpl.ExecuteTemplate(w, "index.gohtml", nil)

}
func results(w http.ResponseWriter, req *http.Request) {

	gen := string(req.FormValue("gender"))
	p10 := string(req.FormValue("perc10"))
	p12 := string(req.FormValue("perc12"))
	s := string(req.FormValue("stream"))
	pg := string(req.FormValue("percgrad"))
	workex := string(req.FormValue("workex"))
	pmba := string(req.FormValue("percmba"))

	csvFile, err := os.Create("test.csv")
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}
	empData := [][]string{
		// {"gender","ssc_p","hsc_p","hsc_s","degree_p","degree_t","workex","mba_p"},
		{gen, p10, p12, s, pg, workex, pmba},
	}
	fmt.Println(empData)
	csvwriter := csv.NewWriter(csvFile)
	for _, empRow := range empData {
		_ = csvwriter.Write(empRow)
	}
	csvwriter.Flush()
	csvFile.Close()
	predData, err := base.ParseCSVToInstances("test.csv", false)
	if err != nil {
		panic(err)
	}

	classificationData, err := base.ParseCSVToInstances("final_data.csv", true)
	if err != nil {
		panic(err)
	}
	trainData, testData := base.InstancesTrainTestSplit(classificationData, 0.5)

	fmt.Println(trainData)
	decTree := trees.NewDecisionTreeClassifier("entropy", -1, []int64{0, 1})
	// knn.NewKnnClassifier("euclidean", "linear", 2)

	// Train Tree
	err = decTree.Fit(trainData)
	if err != nil {
		panic(err)
	}
	// Print out tree for visualization - shows splits and feature and predictions
	fmt.Println(decTree.String())

	// Access Predictions
	classificationPreds := decTree.Predict(trainData)

	// fmt.Println(testData)

	fmt.Println(classificationPreds)

	// Evaluate Accuracy on Test Data
	fmt.Println(decTree.Evaluate(testData))

	fmt.Println("Career Predictions on result")
	pred := decTree.Predict(predData)
	fmt.Println(pred)
	var res string
	if pred[0] == 0 {
		res = "This student is not likely to be placed."
	} else {
		res = "This student is likely to be placed."

	}

	p := map[string]interface{}{"gen": gen,
		"p10": p10, "p12": p12, "s": s, "pg": pg, "workex": workex, "pmba": pmba, "res": res}
	tpl.ExecuteTemplate(w, "results.gohtml", p)
}

func csv_file() {
	csvFile, err := os.Create("test.csv")
	if err != nil {
		log.Fatalf("failed creating file: %s", err)
	}
	empData := [][]string{
		// {"gender","ssc_p","hsc_p","hsc_s","degree_p","degree_t","workex","mba_p"},
		{"1.0", "67.0", "91.0", "1.0", "58.0", "2.0", "0.0", "58.8"},
	}

	csvwriter := csv.NewWriter(csvFile)
	for _, empRow := range empData {
		_ = csvwriter.Write(empRow)
	}
	csvwriter.Flush()
	csvFile.Close()
}

/*func train(){
  classificationData, err := base.ParseCSVToInstances("final_data.csv",true)
	if err != nil {
		panic(err)
	}
	trainData, testData := base.InstancesTrainTestSplit(classificationData, 0.5)

  fmt.Println(trainData)
//decTree := trees.NewDecisionTreeClassifier("entropy", -1, []int64{0,1})
  // knn.NewKnnClassifier("euclidean", "linear", 2)


	// Train Tree
	err = decTree.Fit(trainData)
	if err != nil {
		panic(err)
	}
	// Print out tree for visualization - shows splits and feature and predictions
	fmt.Println(decTree.String())

	// Access Predictions
	classificationPreds := decTree.Predict(trainData)

  // fmt.Println(testData)

	fmt.Println(classificationPreds)

	// Evaluate Accuracy on Test Data
	fmt.Println(decTree.Evaluate(testData))




}*/
