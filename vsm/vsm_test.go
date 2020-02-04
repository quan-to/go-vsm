package vsm

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode"

	"golang.org/x/text/runes"
	"golang.org/x/text/transform"
)

var fromFile = flag.Bool("fromfile", false, `test from files inside "testdata" dir.`)
var fileName = flag.String("filename", "training.json", "name of the file that contains the tests.")

type fileTest struct {
	Docs      []Document `json:"documents"`
	Transform *struct {
		Map *struct {
			Runes string `json:"runes"`
			To    string `json:"to"`
		} `json:"map"`
	} `json:"transform"`
	Tests []struct {
		Query string `json:"query"`
		Want  string `json:"wantClass"`
	} `json:"tests"`
}

// openTestFile reads a file from `name` and returns a file descriptor.
// The file should be in `testdata` dir, as the `name` will be prefixed
// with `testdata/`. It the caller's responsability to close the file when needed.
func openTestFile(name string) (*os.File, error) {
	file, err := os.Open(fmt.Sprintf("testdata%s%s", string(os.PathSeparator), name))
	if err != nil {
		return nil, err
	}

	return file, nil
}

func setupTraining(t *testing.T, vsm *VSM, docs []Document) {
	trainingCh := make(chan Document, len(docs))
	defer close(trainingCh)

	for _, doc := range docs {
		trainingCh <- doc
	}

	trainedCh, errCh := vsm.Train(context.Background(), trainingCh)

	for i := 0; i < len(docs); i++ {
		select {
		case err := <-errCh:
			t.Fatalf("Got error while training: %q.", err)
		case <-trainedCh:
		case <-time.Tick(500 * time.Millisecond):
			t.Fatal("Got training timed out.")
		}
	}
}

func setupTransformer(training fileTest) transform.Transformer {
	var transf transform.Transformer
	if training.Transform != nil {
		var transformers []transform.Transformer

		if m := training.Transform.Map; m != nil {
			newRune := []rune(m.To)
			transformers = append(transformers, runes.Map(func(r rune) rune {
				if strings.ContainsRune(m.Runes, r) && len(newRune) > 0 {
					return newRune[0]
				}
				return r
			}))
		}

		if len(transformers) > 0 {
			transf = transform.Chain(transformers...)
		}
	}

	return transf
}

func TestVSMSearchFromFile(t *testing.T) {
	if !*fromFile {
		t.Skip("Skipping tests loaded from file")
	}

	f, err := openTestFile(*fileName)
	if err != nil {
		t.Fatalf("got error loading test file: 'testdata/%s'.", *fileName)
	}
	defer f.Close()

	var training fileTest

	if err := json.NewDecoder(f).Decode(&training); err != nil {
		t.Fatalf("got error parsing test file 'testdata/%s'.", *fileName)
	}

	for _, tc := range training.Tests {
		t.Run(tc.Query, func(t *testing.T) {
			vsm := New(setupTransformer(training))

			setupTraining(t, vsm, training.Docs)

			doc := vsm.Search(tc.Query)
			if doc == nil {
				t.Fatalf("Got no document found for query: %q.", tc.Query)
			}

			if got := doc.Class; got != tc.Want {
				t.Errorf("Got %q class; want %q.", got, tc.Want)
			}
		})
	}
}

func TestClassificationSearch(t *testing.T) {

	docs := []Document{
		Document{
			Sentence: "Shipment of gold damaged in a fire.",
			Class:    "d1",
		},
		Document{
			Sentence: "Delivery of silver arrived in a silver truck.",
			Class:    "d2",
		},
		Document{
			Sentence: "Shipment-of-gold-arrived in a truck.",
			Class:    "d3",
		},
	}

	testCases := []struct {
		transformer transform.Transformer
		query       string
		want        *Document
	}{
		{
			transformer: nil,
			query:       "gold silver truck.",
			want:        &Document{"Delivery of silver arrived in a silver truck.", "d2"},
		},
		{
			transformer: nil,
			query:       "shipment gold fire.",
			want:        &Document{"Shipment of gold damaged in a fire.", "d1"},
		},
		{
			transformer: runes.Map(func(r rune) rune {
				if unicode.Is(unicode.Hyphen, r) {
					return ' '
				}
				return r
			}),
			query: "shipment gold truck.",
			want:  &Document{"Shipment-of-gold-arrived in a truck.", "d3"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.query, func(t *testing.T) {
			vsm := New(tc.transformer)

			setupTraining(t, vsm, docs)

			got := vsm.Search(tc.query)

			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Got %+v classifier; want %+v.", got, tc.want)
			}
		})
	}
}

func TestClassificationTrainError(t *testing.T) {
	ctx, _ := context.WithDeadline(context.Background(), time.Now())

	vsm := New(nil)

	trainingCh := make(chan Document, 1)
	defer close(trainingCh)

	trainingCh <- Document{}

	_, errCh := vsm.Train(ctx, trainingCh)

	select {
	case err := <-errCh:
		if err == nil {
			t.Error("Got error nil, want not nil.")
		}
	case <-time.Tick(500 * time.Millisecond):
		t.Fatal("error channel timed out")
	}
}
