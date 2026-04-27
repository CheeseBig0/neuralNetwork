public class Neuron {

    // Weights for the two inputs
    double w1, w2;

    // Bias shifts the output left/right
    double bias;


    /**
     * Initializes weights and bias randomly between -1 and 1.
     */
    public Neuron() {
        w1 = Math.random() * 2 - 1;
        w2 = Math.random() * 2 - 1;
        bias = Math.random() * 2 - 1;
    }


    public static double sigmoid(double in){
        return 1 / (1 + Math.exp(-in));
    }

    /**
     * Computes the output of this neuron.
     */
    public double predict(double x1, double x2) {
        double z = w1 * x1 + w2 * x2 + bias;
        return sigmoid(z);
    }

	// We will need the raw output of this node for training purposes later
    public double rawValue(double x1, double x2) {
        return w1 * x1 + w2 * x2 + bias;
    }

    
    /**
     * Updates weights using gradient descent.
     * 
     * grad_w1 = how much w1 contributed to the error
     * lr = learning rate (step size)
     * 
     * We subtract because we want to reduce error.
     */
    public void update(double grad_w1, double grad_w2, double grad_b, double lr) {
        w1 -= lr * grad_w1;
        w2 -= lr * grad_w2;
        bias -= lr * grad_b;
    }
    /**
     * Derivative of sigmoid:
     * Tells us how sensitive the output is to changes in input.
     * 
     * Used during backpropagation.
     */
    public static double sigmoidDerivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }


}
