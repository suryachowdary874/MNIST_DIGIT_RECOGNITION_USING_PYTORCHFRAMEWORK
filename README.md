MNIST Digit Classifier — PyTorch

A neural network built with PyTorch that classifies handwritten digits (0-9) from the MNIST dataset, trained and evaluated end to end.

Overview

This project loads the MNIST dataset, trains a simple feedforward neural network to recognize handwritten digits, and evaluates its accuracy on unseen test data.

Model Architecture
Input (28x28 image)
   → Flatten → 784 values
   → Linear(784, 128)
   → ReLU
   → Linear(128, 10)
   → Output (10 class scores, one per digit 0-9)

The model is a simple fully-connected (feedforward) network with one hidden layer of 128 neurons and a ReLU activation, which allows it to learn non-linear patterns in the pixel data.

Dataset
Source: torchvision.datasets.MNIST
Training set: 60,000 labeled images
Test set: 10,000 labeled images, held out and never used during training
Preprocessing: raw images converted to tensors with pixel values scaled to [0, 1] via transforms.ToTensor()
Training Setup
Component	Choice
Optimizer	Adam, learning rate = 0.001
Loss function	CrossEntropyLoss (standard for multi-class classification)
Batch size	64
Epochs	3
Training Loop

Each batch follows the standard PyTorch training pattern:

python
optimizer.zero_grad()      # clear gradients from the previous step
preds = model(images)       # forward pass
loss = loss_fn(preds, labels)
loss.backward()              # compute gradients via autograd
optimizer.step()             # update weights
Evaluation

After training, the model is switched to evaluation mode (model.eval()) and run on the held-out test set with gradient tracking disabled (torch.no_grad()). For each image, the predicted digit is taken as the class with the highest output score (argmax), and accuracy is computed as the percentage of correct predictions across all 10,000 test images.

Results
Training loss decreased steadily across epochs (approx. 0.35 → 0.16 → 0.10)
Final test accuracy: ~95% on unseen data
How to Run
bash
pip install torch torchvision
python mnist_classifier.py

Or open in Google Colab — no local installation required.

What This Project Demonstrates
Building a custom neural network using nn.Module
Working with real image data via torchvision
Understanding of autograd, gradient descent, and the standard PyTorch training loop
The role of activation functions (ReLU) in enabling non-linear learning
Correct train/test separation to evaluate generalization rather than memorization
Possible Improvements
Replace fully-connected layers with a CNN (nn.Conv2d) for better accuracy on image data
Add dropout or batch normalization to reduce overfitting
Tune learning rate and batch size, or add a learning rate scheduler
Track validation accuracy per epoch, not just training loss
