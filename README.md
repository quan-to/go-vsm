# go-vsm

[![GoDoc](https://godoc.org/github.com/quan-to/go-vsm?status.svg)](https://godoc.org/github.com/quan-to/go-vsm/vsm)

Vector Space Model implementation in Go.

This package provides document search besed on the algebraic [Vector Space Model](https://en.wikipedia.org/wiki/Vector_space_model). The weighting scheme used is the [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

## Usage

```go
import "github.com/quan-to/go-vsm/vsm"
```

Construct a VSM object and use the methods of the VSM object for training:

```go
vsm := New(nil)

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
                Sentence: "Shipment of gold arrived in a truck.",
                Class:    "d3",
        },
        ...,
}


// Statically training
for _, doc := range docs {
        if err := vsm.StaticTraining(doc); err != nil {
                // Error occurred during training.
        }
}
```

### Dynamic Training

Static training is executed once, and for most cases it's enough:

```go
docs := []Document{
        Document{
                Sentence: "Shipment of gold damaged in a fire.",
                Class:    "d1",
        },
        Document{
                Sentence: "Shipment of gold arrived in a truck.",
                Class:    "d3",
        },
}

vsm := New(nil)

for _, doc := range docs {
        err := vsm.StaticTraining(doc)
        fmt.Println(err)
}

```

But if you've got a stream of data and need a more reactive behaviour for the training process, the dynamic training might be the best choice:

```go
docCh := make(chan Document)

go func() {
        defer close(docCh)

        // Loads document from some source dynamically
        // and sends it to the training channel.
        docCh <- Document{
                Sentence: "Delivery of silver arrived in a silver truck.",
                Class:    "d2",
        }
}()

trainCh := vsm.DynamicTraining(context.Background(), docCh)

// Checks if error occurred during the training process.
for {
        res, ok := <-trainCh
        // trainCh closed. All train data was consumed.
        if !ok {
                break
        }

        if res.Err != nil {
                // Handles error.
        }
}
```

### Search

Search applies the Vector Space Model to compare the deviation of angles between each document vector and the query vector.

```
doc, err := vsm.Search("gold silver truck.")

fmt.Println(doc.Class, err) // d2
```

## Testing

Go to `vsm` folder and run:

```bash
go test -v -cover
```

This package provides a way of testing through file:

```bash
go test -v -fromfile
```

The `-fromfile` flag tells the test to run tests over the `testdata/training.json` file. 

If you want to specify another testing file:

```bash
go test -v -fromfile -filename="training-2.json"
```

The `-filename` flag should point to a file inside the `testdata` folder. See the [training.json](vsm/testdata/training.json) file for details on its format.

## LICENSE - MIT

see [LICENSE](LICENSE)
