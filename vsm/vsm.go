package vsm

import (
	"context"
	"math"
	"strings"
	"sync"
	"sync/atomic"

	"golang.org/x/text/transform"
)

type Document struct {
	Sentence string `json:"sentence"`
	Class    string `json:"class"`
}

type term struct {
	// DocsSeen holds the number of times
	// the term has been seen in document.
	// But it is only counted once even
	// if it's in a document multiple times.
	docsSeen uint64
}

type terms struct {
	mu    sync.RWMutex
	terms map[string]term
}

func (t *terms) Get(term string) (term, bool) {
	t.mu.RLock()
	res, ok := t.terms[term]
	t.mu.RUnlock()
	return res, ok
}

func (t *terms) Set(k string, v term) {
	t.mu.Lock()
	t.terms[k] = v
	t.mu.Unlock()
}

type document struct {
	Document

	termFreq map[string]uint64
}

type VSM struct {
	terms *terms

	mu   sync.RWMutex
	docs []document

	docsCount uint64

	transformer transform.Transformer
}

func New(t transform.Transformer) *VSM {
	vsm := &VSM{
		terms:       &terms{sync.RWMutex{}, make(map[string]term)},
		mu:          sync.RWMutex{},
		docs:        []document{},
		transformer: t,
	}

	return vsm
}

type TrainResult struct {
	Doc Document
	Err error
}

func (v *VSM) Train(ctx context.Context, docCh <-chan Document) <-chan TrainResult {
	trainCh := make(chan TrainResult)

	go func() {
		defer close(trainCh)

		for {
			select {
			case doc := <-docCh:
				trainCh <- TrainResult{Doc: doc, Err: v.train(doc)}
			case <-ctx.Done():
				trainCh <- TrainResult{Err: ctx.Err()}
				return
			}
		}
	}()

	return trainCh
}

func (v *VSM) Search(query string) (*Document, error) {
	queryDoc := document{termFreq: make(map[string]uint64)}

	query, err := v.sanitize(query)
	if err != nil {
		return nil, err
	}

	for _, term := range strings.Split(query, " ") {
		t := strings.ToLower(strings.TrimSpace(term))

		queryDoc.termFreq[t]++
	}

	totalDocs := atomic.LoadUint64(&v.docsCount)

	var querySum float64
	for trm, freq := range queryDoc.termFreq {
		t, ok := v.terms.Get(trm)
		if !ok {
			continue
		}

		idf := math.Log(float64(totalDocs) / float64(t.docsSeen))

		weight := float64(freq) * idf

		querySum += math.Pow(weight, 2)
	}

	queryMag := math.Sqrt(querySum)

	var foundDoc *Document

	var maxSim float64
	v.mu.RLock()
	for _, doc := range v.docs {
		var docSum float64
		var coeff float64

		for trm, freq := range doc.termFreq {
			t, _ := v.terms.Get(trm)

			idf := math.Log(float64(totalDocs) / float64(t.docsSeen))

			weight := float64(freq) * idf

			queryTermWeight := float64(queryDoc.termFreq[trm]) * idf

			coeff += weight * queryTermWeight

			docSum += math.Pow(float64(weight), 2)
		}

		docMag := math.Sqrt(docSum)

		sim := coeff / (docMag * queryMag)
		if sim > maxSim {
			foundDoc = &Document{Sentence: doc.Sentence, Class: doc.Class}
			maxSim = sim
		}
	}
	v.mu.RUnlock()

	return foundDoc, nil
}

func (v *VSM) train(dc Document) error {
	doc := document{Document: dc, termFreq: make(map[string]uint64)}

	seenTerms := make(map[string]struct{})

	sentence, err := v.sanitize(dc.Sentence)
	if err != nil {
		return err
	}

	for _, trm := range strings.Split(sentence, " ") {
		t := strings.ToLower(strings.TrimSpace(trm))

		if _, ok := v.terms.Get(t); !ok {
			v.terms.Set(t, term{})
		}

		doc.termFreq[t]++

		seenTerms[t] = struct{}{}
	}

	for trm := range seenTerms {
		t, _ := v.terms.Get(trm)
		t.docsSeen++
		v.terms.Set(trm, t)
	}

	v.mu.Lock()
	v.docs = append(v.docs, doc)
	v.mu.Unlock()

	atomic.StoreUint64(&v.docsCount, uint64(len(v.docs)))

	return nil
}

func (v *VSM) sanitize(sentence string) (string, error) {
	if v.transformer != nil {
		sanitized, _, err := transform.String(v.transformer, sentence)
		if err != nil {
			return sentence, err
		}

		return sanitized, nil
	}

	return sentence, nil
}
