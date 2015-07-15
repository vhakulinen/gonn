package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/vhakulinen/gonn"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	// e.g. { 3, 2, 1}
	topology := []int{2, 4, 1}
	myNet := gonn.NewNetwork(topology)

	for i, layer := range myNet.Layers {
		fmt.Printf("Layer %d has %d neurons\n", i, len(layer))
	}

	for n := 0; n < 1000; n++ {
		input := []float64{
			float64(rand.Intn(2)),
			float64(rand.Intn(2)),
		}
		myNet.FeedForward(input)
		target := float64(int(input[0]) ^ int(input[1]))
		myNet.BackProp([]float64{target})
		fmt.Printf("Inputs: %v - Output: %v Target: %v\n",
			input, myNet.GetResults(), target)
	}
}
