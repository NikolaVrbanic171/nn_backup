import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_dimension, output_dimension, mean=0.3, deviation=0.3):
        weights = np.random.normal(loc=mean, scale=deviation, size=(input_dimension, output_dimension))
        bias = np.ones((1, output_dimension))
        layer = np.concatenate((bias, weights), axis=0)
        self.layers.append(layer)

    def forward_pass(self, inputs):
        import copy
        izlazi = []
        izlazi.append(inputs)
        for layer in self.layers:
            weights = layer[1:, :]
            bias = layer[0, :]
            inputs = np.array(sigmoid(inputs @ weights - bias))
            izlazi.append(copy.deepcopy(inputs.T))
        return inputs, izlazi #Izlaz i lista svih izlaza

    def back_prop(self, outputs, real_output, batch_size, learning_rate):
        import copy
        error_matrice = []
        bias_correction = []

        output_error = outputs[-1] - np.array(real_output).T #.reshape(-1,batch_size)
        delta_output = output_error*dsigmoid(outputs[-1])
        output_weights_error = delta_output@outputs[-2].T

        error_matrice.append(copy.deepcopy(output_weights_error))
        bias_correction.append(copy.deepcopy(-delta_output))

        for i, output in enumerate(reversed(outputs[1:-1]), 2):
            out_err_proslog = -np.array(bias_correction[-1]) #zadnji dodan u listu je prijasnji
            weights_proslog = self.layers[-(i-1)][1:, :]
            out_error = weights_proslog@out_err_proslog
            delta_output = out_error*dsigmoid(output)
            output_weights_error = delta_output@outputs[-(i+1)].reshape(batch_size, -1)

            error_matrice.append(copy.deepcopy(output_weights_error))
            bias_correction.append(copy.deepcopy(-delta_output))

        for i in range(len(self.layers)):
            stari_bias = self.layers[-(1+i)][0, :]
            stari_weights = self.layers[-(1+i)][1:, :]
            korekcija_biasa = np.sum(bias_correction[i], axis=1)
            updated_bias = np.array(stari_bias - learning_rate*korekcija_biasa)
            updated_weights =stari_weights - learning_rate*error_matrice[i].T
            self.layers[-(i+1)] = np.concatenate((updated_bias.reshape(1,updated_bias.shape[0]), updated_weights), axis=0)


    def train(self, broj_epoha, batch_size, input_data, output_data, learning_rate, ciljna_preciznost):
        broj_podataka = input_data.shape[0]
        br_koraka = int(broj_podataka/batch_size)
        for i in range(broj_epoha):
            for j in range(br_koraka):
                dg = j*batch_size
                gg = (1+j)*batch_size
                output, all_outputs = self.forward_pass(input_data[dg: gg])
                self.back_prop(outputs=all_outputs, real_output=output_data[dg: gg],batch_size=batch_size, learning_rate=learning_rate)
            mse = np.mean((self.forward_pass(input_data)[0]-output_data.reshape(-1,self.forward_pass(input_data)[0].shape[1]))**2)
            if mse<ciljna_preciznost:
                print("Istrenirano")
                break
            #if i%1000 == 0:
                #learning_rate /= 2
            if i%100 == 0:
                print(i, ". ", mse)

# nn = NeuralNetwork()
# nn.add_layer(2, 4)
# nn.add_layer(4, 4)
# nn.add_layer(4, 1)
# realoutput = np.array([0, 1, 1, 0])
# nn.train(10000, 2, np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), realoutput, 1, 1e-6)
# print(nn.forward_pass(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))[0])
