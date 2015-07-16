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
	return 1.0 - x*x
}

type neuronConnection struct {
	Weight      float64
	DeltaWeight float64
}

// Neuron object
type Neuron struct {
	outputVal     float64
	outputWeights []neuronConnection
	myIndex       int
	gradient      float64

	//eta   float64 // [0.0..1.0] overall net training rate
	//Alpha float64 // [0.0..n] multiplier of last weight chagne (momentum)
}

// NewNeuron intializes new neuron object
func NewNeuron(numOutputs, myIndex int) *Neuron {
	n := new(Neuron)
	// c for connections
	for c := 0; c < numOutputs; c++ {
		n.outputWeights = append(n.outputWeights, *new(neuronConnection))
		n.outputWeights[len(n.outputWeights)-1].Weight = randomWeight()
	}
	n.myIndex = myIndex
	return n
}

// FeedForward does the math magic to its self
func (n *Neuron) FeedForward(prevLayer *Layer) {
	var sum float64

	// Sum the previous layer's outputs (which are our inputs)
	// Include the bias node from the previous layer

	for i := 0; i < len(*prevLayer); i++ {
		sum += (*prevLayer)[i].outputVal *
			(*prevLayer)[i].outputWeights[n.myIndex].Weight
	}

	n.outputVal = transferFunction(sum)
}

func (n Neuron) sumDOW(nextLayer *Layer) float64 {
	var sum float64

	for i := 0; i < len(*nextLayer)-1; i++ {
		sum += n.outputWeights[i].Weight * (*nextLayer)[i].gradient
	}

	return sum
}

func (n *Neuron) calculateOutputGradients(targetVal float64) {
	delta := targetVal - n.outputVal
	n.gradient = delta * transferFunctionDerivative(n.outputVal)
}

func (n *Neuron) calculateHiddenGradients(nextLayer *Layer) {
	dow := n.sumDOW(nextLayer)
	n.gradient = dow * transferFunctionDerivative(n.outputVal)
}

func (n *Neuron) updateInputWeights(prevLayer *Layer) {
	// The weights to be updated are in the Conneciton container
	// int the neurons in the preceding layer
	for i := 0; i < len(*prevLayer); i++ {
		neuron := &(*prevLayer)[i]
		oldDeltaWeight := neuron.outputWeights[n.myIndex].DeltaWeight

		newDeltaWeight :=
			// Individual input, magnified by the gradient and train rate:
			Eta*neuron.outputVal*n.gradient +
				// Also add momentun = a fraction of the previous delta wieght
				Alpha*oldDeltaWeight
		neuron.outputWeights[n.myIndex].DeltaWeight = newDeltaWeight
		neuron.outputWeights[n.myIndex].Weight += newDeltaWeight
	}
}

// Layer is just array of neurons
type Layer []Neuron

// NeuralNetwork holds all the data of the network
type NeuralNetwork struct {
	Layers                            []Layer // layers[layerNum][neuronNum]
	err                               float64
	recentAverageError                float64
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
		(*layer)[len(*layer)-1].outputVal = 1.0
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
		n.Layers[0][i].outputVal = inputVals[i]
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
	n.err = 0.0

	for i := 0; i < len(*outputLayer)-1; i++ {
		delta := targetVals[i] - (*outputLayer)[i].outputVal
		n.err += delta * delta
	}
	n.err /= float64(len(*outputLayer)) - 1.0 // get average error squared
	n.err = math.Sqrt(n.err)                  // RMS

	// Implements a recent average measurement
	n.recentAverageError =
		(n.recentAverageError*n.RecentAverageErrorSmoothingFactor + n.err) /
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
	return n.recentAverageError
}

// GetResults returns results from all output neurons as a string
func (n *NeuralNetwork) GetResults() string {
	var out string
	for i, outputNeuron := range n.Layers[len(n.Layers)-1] {
		if i == len(n.Layers[len(n.Layers)-1])-1 {
			// Ignore bias
			continue
		}
		out += fmt.Sprintf("%f ", outputNeuron.outputVal)
	}
	return out
}
