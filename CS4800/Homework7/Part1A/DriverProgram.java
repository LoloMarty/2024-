public class DriverProgram {
    public static void main(String[] args) {
        CharString document = Disk.getDocument();

        document.save("H", "Arial", "Red", 12);
        document.save("e", "Calibri", "Blue", 14);
        document.save("l", "Verdana", "Black", 16);
        document.save("l", "Roboto", "White", 12);
        document.save("o", "Arial", "Red", 12);
        document.save("W", "Arial", "Red", 12);
        document.save("o", "Calibri", "Blue", 14);
        document.save("r", "Verdana", "Black", 16);
        document.save("l", "Roboto", "White", 12);
        document.save("d", "Arial", "Red", 12);
        document.save("C", "Arial", "Red", 12);
        document.save("S", "Calibri", "Blue", 14);
        document.save("5", "Verdana", "Black", 16);
        document.save("8", "Roboto", "White", 12);
        document.save("0", "Arial", "Red", 12);
        document.save("0", "Arial", "Red", 12);

        document.load();

    }
}
