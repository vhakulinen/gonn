// Package gonn is a port from this: http://inkdrop.net/dave/docs/neural-net-tutorial.cpp
package gonn

import (
	"fmt"
	"log"
	"math"
	"math/rand"
)

var (
	// Eta [0.0..1.0] overall net training rate
	Eta = 0.15
	// Alpha [0.0..1.0] multiplier of last weight chagne (momentum)
	Alpha = 0.5
)

func randomWeight() float64 {
	return rand.Float64()
}

func transferFunction(x float64) float64 {
	// tanh - output range [-1.0..1.0]
	//return 1.0 / (1.0 + math.Exp(-x))
	return math.Tanh(x)
}

func transferFunctionDerivative(x float64) float64 {
	// tanh derivative
	// not the actual formula
	//return 1.0
	//return (1.0 / (1.0 + math.Exp(-x))) - (1.0+math.Exp(-x))*(1.0+math.Exp(-x))
	return 1.0 - x*x
}

// NeuronConnection is made between layers and their neurons
type NeuronConnection struct {
	Weight      float64
	DeltaWeight float64
}

// Neuron object
type Neuron struct {
	OutputVal     float64
	OutputWeights []NeuronConnection
	MyIndex       int
	Gradient      float64

	//eta   float64 // [0.0..1.0] overall net training rate
	//Alpha float64 // [0.0..n] multiplier of last weight chagne (momentum)
}

// NewNeuron intializes new neuron object
func NewNeuron(numOutputs, myIndex int) *Neuron {
	n := new(Neuron)
	// c for connections
	for c := 0; c < numOutputs; c++ {
		n.OutputWeights = append(n.OutputWeights, *new(NeuronConnection))
		n.OutputWeights[len(n.OutputWeights)-1].Weight = randomWeight()
	}
	n.MyIndex = myIndex
	return n
}

// FeedForward does the math magic to its self
func (n *Neuron) FeedForward(prevLayer *Layer) {
	var sum float64

	// Sum the previous layer's outputs (which are our inputs)
	// Include the bias node from the previous layer

	for i := 0; i < len(*prevLayer); i++ {
		sum += (*prevLayer)[i].OutputVal *
			(*prevLayer)[i].OutputWeights[n.MyIndex].Weight
	}

	n.OutputVal = transferFunction(sum)
}

func (n Neuron) sumDOW(nextLayer *Layer) float64 {
	var sum float64

	for i := 0; i < len(*nextLayer)-1; i++ {
		sum += n.OutputWeights[i].Weight * (*nextLayer)[i].Gradient
	}

	return sum
}

func (n *Neuron) calculateOutputGradients(targetVal float64) {
	delta := targetVal - n.OutputVal
	n.Gradient = delta * transferFunctionDerivative(n.OutputVal)
}

func (n *Neuron) calculateHiddenGradients(nextLayer *Layer) {
	dow := n.sumDOW(nextLayer)
	n.Gradient = dow * transferFunctionDerivative(n.OutputVal)
}

func (n *Neuron) updateInputWeights(prevLayer *Layer) {
	// The weights to be updated are in the Conneciton container
	// int the neurons in the preceding layer
	for i := 0; i < len(*prevLayer); i++ {
		neuron := &(*prevLayer)[i]
		oldDeltaWeight := neuron.OutputWeights[n.MyIndex].DeltaWeight

		newDeltaWeight :=
			// Individual input, magnified by the gradient and train rate:
			Eta*neuron.OutputVal*n.Gradient +
				// Also add momentun = a fraction of the previous delta wieght
				Alpha*oldDeltaWeight
		neuron.OutputWeights[n.MyIndex].DeltaWeight = newDeltaWeight
		neuron.OutputWeights[n.MyIndex].Weight += newDeltaWeight
	}
}

// Layer is just array of neurons
type Layer []Neuron

// NeuralNetwork holds all the data of the network
type NeuralNetwork struct {
	Layers                            []Layer // layers[layerNum][neuronNum]
	Err                               float64
	RecentAverageError                float64
	RecentAverageErrorSmoothingFactor float64
}

// NewNetwork initializes new network
func NewNetwork(topology []int) *NeuralNetwork {
	n := new(NeuralNetwork)
	// Number of training smaples to average over
	//n.RecentAverageErrorSmoothingFactor = 112.0

	for layerNum := 0; layerNum < len(topology); layerNum++ {
		n.Layers = append(n.Layers, *new(Layer))
		var numOutputs int
		if layerNum == len(topology)-1 {
			numOutputs = 0
		} else {
			numOutputs = topology[layerNum+1]
		}

		// We have made new layer, now fill in its neurons
		// and a bias neuron
		for neuronNum := 0; neuronNum <= topology[layerNum]; neuronNum++ {
			n.Layers[len(n.Layers)-1] =
				append(n.Layers[len(n.Layers)-1],
					*NewNeuron(numOutputs, neuronNum))
		}

		// Force the bias node's output value to 1.0. It's the last neuron
		// created above
		layer := &n.Layers[layerNum]
		(*layer)[len(*layer)-1].OutputVal = 1.0
	}
	return n
}

// FeedForward takes inputs
func (n *NeuralNetwork) FeedForward(inputVals []float64) {
	// Ignore bias
	if len(inputVals) > len(n.Layers[0])-1 {
		log.Fatalf("Length if inputsVals must be the same as length of"+
			" the first layer (%d != %d)", len(inputVals), len(n.Layers[0]))
	}

	// assign the input values into the input neurons
	for i := 0; i < len(inputVals); i++ {
		n.Layers[0][i].OutputVal = inputVals[i]
	}

	// Forward propagate
	for layerNum := 1; layerNum < len(n.Layers); layerNum++ {
		prevLayer := &n.Layers[layerNum-1]
		for i := 0; i < len(n.Layers[layerNum])-1; i++ {
			n.Layers[layerNum][i].FeedForward(prevLayer)
		}
	}
}

// BackProp does the backpropagation (this is where the net learns)
func (n *NeuralNetwork) BackProp(targetVals []float64) {
	// Calculate overall net error (RMS of output errors)
	// RMS = "Root Mean Square Error"
	outputLayer := &n.Layers[len(n.Layers)-1]
	n.Err = 0.0

	for i := 0; i < len(*outputLayer)-1; i++ {
		delta := targetVals[i] - (*outputLayer)[i].OutputVal
		n.Err += delta * delta
	}
	n.Err /= float64(len(*outputLayer)) - 1.0 // get average error squared
	n.Err = math.Sqrt(n.Err)                  // RMS

	//if n.Err > 2.0 {
	//n.Err = 1.0
	//}

	// Implements a recent average measurement
	n.RecentAverageError =
		(n.RecentAverageError*n.RecentAverageErrorSmoothingFactor + n.Err) /
			(n.RecentAverageErrorSmoothingFactor + 1.0)

	// Calculate output layer gradiants
	for i := 0; i < len(*outputLayer)-1; i++ {
		(*outputLayer)[i].calculateOutputGradients(targetVals[i])
	}

	// Calculate gradients on hidden layers
	for layerNum := len(n.Layers) - 2; layerNum > 0; layerNum-- {
		hiddenLayer := &n.Layers[layerNum]
		nextLayer := &n.Layers[layerNum+1]

		for i := 0; i < len(*hiddenLayer); i++ {
			(*hiddenLayer)[i].calculateHiddenGradients(nextLayer)
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights

	for layerNum := len(n.Layers) - 1; layerNum > 0; layerNum-- {
		layer := &n.Layers[layerNum]
		prevLayer := &n.Layers[layerNum-1]

		for i := 0; i < len(*layer)-1; i++ {
			(*layer)[i].updateInputWeights(prevLayer)
		}
	}
}

// GetAverageError return recentAvarageError value
func (n *NeuralNetwork) GetAverageError() float64 {
	return n.RecentAverageError
}

// GetResults returns results from all output neurons as a string
func (n *NeuralNetwork) GetResults() string {
	var out string
	for i, outputNeuron := range n.Layers[len(n.Layers)-1] {
		if i == len(n.Layers[len(n.Layers)-1])-1 {
			// Ignore bias
			continue
		}
		out += fmt.Sprintf("%f ", outputNeuron.OutputVal)
	}
	return out
}
