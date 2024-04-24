public class TextEditor {
    public static void main(String[] args) {
        // Creating and editing documents with different character properties
        Document document = new Document();
        document.addCharacter('H', "Arial", "Red", 12);
        document.addCharacter('e', "Calibri", "Blue", 14);
        document.addCharacter('l', "Verdana", "Black", 16);
        document.addCharacter('l', "Arial", "Red", 12); // Reusing existing properties
        document.addCharacter('o', "Calibri", "Blue", 14); // Reusing existing properties

        // Save and load document
        document.save("HelloWorldCS5800.txt");
        Document loadedDocument = Document.load("HelloWorldCS5800.txt");
        if (loadedDocument != null) {
            loadedDocument.print();
        }
    }
}