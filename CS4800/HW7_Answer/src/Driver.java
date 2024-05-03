import java.io.IOException;

public class Driver {

    public static void main(String[] args) throws IOException {

        System.out.println("Create a new Document");

        Document testDocument = new Document();

        // HelloWorldCS5800

        testDocument.addCharacter('H', "Arial", "Black", 16);
        testDocument.addCharacter('e', "Arial", "Black", 14);
        testDocument.addCharacter('l', "Arial", "Black", 14);
        testDocument.addCharacter('l', "Arial", "Black", 14);
        testDocument.addCharacter('o', "Arial", "Black", 14);
        testDocument.addCharacter('W', "Arial", "Black", 16);
        testDocument.addCharacter('o', "Arial", "Black", 14);
        testDocument.addCharacter('r', "Arial", "Black", 14);
        testDocument.addCharacter('l', "Arial", "Black", 14);
        testDocument.addCharacter('d', "Arial", "Black", 14);
        testDocument.addCharacter('C', "Verdana", "Blue", 18);
        testDocument.addCharacter('S', "Verdana", "Blue", 18);
        testDocument.addCharacter('5', "Verdana", "Red", 12);
        testDocument.addCharacter('8', "Verdana", "Red", 12);
        testDocument.addCharacter('0', "Verdana", "Red", 12);
        testDocument.addCharacter('0', "Verdana", "Red", 12);

        try {
            testDocument.saveToFile("test.txt");
            System.out.println("Saved Document as test.txt");
        } catch (IOException e) {
            System.out.println("Failed on Save");
        }

        Document readDocumentVerify = new Document();

        try {
            readDocumentVerify.loadFromFile("test.txt");
            System.out.println("Loaded Document from test.txt");
        } catch (IOException e) {
            System.out.println("Failed on Load");
        }

        // H - modify properties
        readDocumentVerify.editCharacterProperties(0, "Calibri", "Blue", 99);
        // W - modify properties
        readDocumentVerify.editCharacterProperties(5, "Calibri", "Blue", 99);

        try {
            readDocumentVerify.saveToFile("verify.txt");
            System.out.println("Saved Modified Document as verify.txt");
        } catch (IOException e) {
            System.out.println("Failed on Verify Save");
        }
    }
}
