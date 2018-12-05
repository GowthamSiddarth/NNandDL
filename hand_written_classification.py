import mnist_loader, ann

train_data, validation_data, test_data = mnist_loader.load_data_wrapper("data/mnist.pkl.gz")
network = ann.Network([784, 1000, 10])
learning_rate, epochs, mini_batch_size = 3, 30, 10
network.stochastic_gradient_descent(train_data, learning_rate, epochs, mini_batch_size, test_data)
