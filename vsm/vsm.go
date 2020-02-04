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

type Term struct {
	// DocsSeen holds the number of times
	// the term has been seen in document.
	// But it is only counted once even
	// if it's in a document multiple times.
	DocsSeen uint64
}

type terms struct {
	sync.RWMutex
	terms map[string]Term
}

func (t *terms) Get(term string) (Term, bool) {
	t.RLock()
	res, ok := t.terms[term]
	t.RUnlock()
	return res, ok
}

func (t *terms) Set(k string, v Term) {
	t.Lock()
	t.terms[k] = v
	t.Unlock()
}

type document struct {
	Document

	TermFreq map[string]uint64
}

type VSM struct {
	terms *terms

	sync.RWMutex
	docs []document

	DocsCount uint64

	Transformer transform.Transformer
}

func New(t transform.Transformer) *VSM {
	vsm := &VSM{
		terms:       &terms{sync.RWMutex{}, make(map[string]Term)},
		RWMutex:     sync.RWMutex{},
		docs:        []document{},
		Transformer: t,
	}

	return vsm
}

type Trained struct {
	Doc Document
}

func (v *VSM) Train(ctx context.Context, docCh <-chan Document) (<-chan Trained, <-chan error) {
	errCh := make(chan error)
	trainedCh := make(chan Trained)

	go func() {
		defer close(errCh)
		defer close(trainedCh)

		for dc := range docCh {
			v.train(dc)

			select {
			case trainedCh <- Trained{dc}:
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}
	}()
	return trainedCh, errCh
}

func (v *VSM) train(dc Document) {
	doc := document{Document: dc, TermFreq: make(map[string]uint64)}

	seenTerms := make(map[string]struct{})

	sentence := dc.Sentence
	if v.Transformer != nil {
		sentence, _, _ = transform.String(v.Transformer, sentence)
	}

	for _, term := range strings.Split(sentence, " ") {
		t := strings.ToLower(strings.TrimSpace(term))

		if _, ok := v.terms.Get(t); !ok {
			v.terms.Set(t, Term{})
		}

		doc.TermFreq[t]++

		seenTerms[t] = struct{}{}
	}

	for term := range seenTerms {
		t, _ := v.terms.Get(term)
		t.DocsSeen++
		v.terms.Set(term, t)
	}

	v.Lock()
	v.docs = append(v.docs, doc)
	v.Unlock()

	atomic.StoreUint64(&v.DocsCount, uint64(len(v.docs)))
}

func (v *VSM) Search(query string) *Document {
	queryDoc := document{TermFreq: make(map[string]uint64)}

	if v.Transformer != nil {
		query, _, _ = transform.String(v.Transformer, query)
	}

	for _, term := range strings.Split(query, " ") {
		t := strings.ToLower(strings.TrimSpace(term))

		queryDoc.TermFreq[t]++
	}

	totalDocs := atomic.LoadUint64(&v.DocsCount)

	var querySum float64
	for term, freq := range queryDoc.TermFreq {
		t, ok := v.terms.Get(term)
		if !ok {
			continue
		}

		idf := math.Log(float64(totalDocs) / float64(t.DocsSeen))

		weight := float64(freq) * idf

		querySum += math.Pow(weight, 2)
	}

	queryMag := math.Sqrt(querySum)

	var foundDoc *Document

	var maxSim float64
	v.RLock()
	for _, doc := range v.docs {
		var docSum float64
		var coeff float64

		for term, freq := range doc.TermFreq {
			t, _ := v.terms.Get(term)

			idf := math.Log(float64(totalDocs) / float64(t.DocsSeen))

			weight := float64(freq) * idf

			queryTermWeight := float64(queryDoc.TermFreq[term]) * idf

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
	v.RUnlock()

	return foundDoc
}
