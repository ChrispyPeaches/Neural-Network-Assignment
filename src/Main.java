public class Main {
    public static void main(String[] args) {
        var network = new NeuralNetwork(new int[]{4, 3, 2}, 2);
        network.TrainNetwork(10, 2);
    }
}
