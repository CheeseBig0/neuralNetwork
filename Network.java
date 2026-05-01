import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Network {

    public static void main(String[] args) {   
 	    List<List<String>> Alldata = readCSV("/workspaces/neuralNetwork/Passwords.csv");
        ArrayList<double[]> data = new ArrayList<double[]>();
        ArrayList<Double> answers = new ArrayList<Double>();

        for(int i=1; i<Alldata.size(); i++) {
            List<String> selected = Alldata.get(i);
            double[] tempData = {0,0};

            //give the password strength out of 10 (thats what code.org said the limit was)
            tempData[0] = Double.parseDouble(selected.get(6)) / 10;
            //give password length with limit of 10 characters
            tempData[1] = ( (double) selected.get(2).length() ) / 10;

            String timeText = selected.get(5);

            //check if the password crack time is at least 1 day
            if( timeText.matches("days") || (timeText.matches("months")) || (timeText.matches("years")) ) {
                answers.add(1.0);
            } else {
                answers.add(0.0);
            }

            data.add(tempData);

            //System.out.println("Strength: " + selected.get(6));
        }
        //System.out.println(data);

        Network network = new Network();
        network.train(data, answers);

        
        System.out.println("Should be yes "+network.predict(0.7, 0.9));

        System.out.println("Should be no "+network.predict(0.1, 0.3));
    }

    Neuron first = new Neuron();
    Neuron second = new Neuron();
    Neuron out = new Neuron();

    /*
     * Data flows:
     * inputs → hidden layer → output layer
     */
    public double predict(double x1, double x2) {

        // Pass inputs into hidden neurons
        double firstOut = first.predict(x1, x2);
        double secondOut = second.predict(x1, x2);

        // Hidden outputs become inputs to final neuron
        return out.predict(firstOut, secondOut);
    }

    /**
     * TRAINING FUNCTION
     * 
     * Uses gradient descent + backpropagation
     * to adjust weights and reduce error.
     */
    public void train(ArrayList<double[]> data, ArrayList<Double> answers) {

        double lr = 0.1; // learning rate

        // Repeat many times so the network can improve
        for (int epoch = 0; epoch < 10000; epoch++) {

            double totalLoss = 0;

            // Go through each training example
            for (int i = 0; i < data.size(); i++) {

                double x1 = data.get(i)[0];
                double x2 = data.get(i)[1]; 
                double correctAnswer = answers.get(i);   

                // =========================
                // 1. FORWARD PASS
                // =========================

                // Hidden neuron 1
                double z1 = first.rawValue(x1, x2);     
                double firstOut = first.predict(x1, x2); 

                // Hidden neuron 2
                double z2 =  second.rawValue(x1, x2); 
                double secondOut = second.predict(x1, x2);

                // Output neuron
                double z3 =  out.rawValue(firstOut, secondOut); 
                double pred = out.predict(firstOut, secondOut);

                // =========================
                // 2. LOSS (error)
                // =========================

                double error = pred - correctAnswer;

                // Mean Squared Error contribution
                totalLoss += error * error;

                // =========================
                // 3. BACKPROPAGATION
                // =========================
                // We compute how much each weight contributed to the error.

                // ----- Output neuron -----

                double predictionLoss = 2 * error;
                double predDerivitive = Neuron.sigmoidDerivative(z3);

                // Gradients for output neuron weights
                double grad_out_w1 = predictionLoss * predDerivitive * firstOut;
                double grad_out_w2 = predictionLoss * predDerivitive * secondOut;
                double grad_out_b  = predictionLoss * predDerivitive;

                // ----- Hidden neurons -----
                // Activation derivatives
                double dz1 = Neuron.sigmoidDerivative(z1);
                double dz2 = Neuron.sigmoidDerivative(z2);

                // Gradients for hidden neuron 1
                double grad_h1_w1 = predictionLoss * predDerivitive * out.w1 * dz1 * x1;
                double grad_h1_w2 = predictionLoss * predDerivitive * out.w1 * dz1 * x2;
                double grad_h1_b  = predictionLoss * predDerivitive * out.w1 * dz1;

                // Gradients for hidden neuron 2
                double grad_h2_w1 = predictionLoss * predDerivitive * out.w2 * dz2 * x1;
                double grad_h2_w2 = predictionLoss * predDerivitive * out.w2 * dz2 * x2;
                double grad_h2_b  = predictionLoss * predDerivitive * out.w2 * dz2;

                // =========================
                // 4. UPDATE WEIGHTS
                // =========================
                // Each neuron adjusts its own weights

                out.update(grad_out_w1, grad_out_w2, grad_out_b, lr);

                first.update(grad_h1_w1, grad_h1_w2, grad_h1_b, lr);
                second.update(grad_h2_w1, grad_h2_w2, grad_h2_b, lr);
            }

            // Print progress every 1000 epochs
            if (epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch +
                        " Loss: " + (totalLoss / data.size()));
            }
        }
    }

    /**
     * Reads CSV file into a 2D list of strings
     */
    public static List<List<String>> readCSV(String filePath) {
        List<List<String>> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                data.add(Arrays.asList(line.split(",")));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return data;
    }
    

}
