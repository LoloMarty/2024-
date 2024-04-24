class Document {
    private StringBuilder content = new StringBuilder();

    public void addCharacter(char c, String font, String color, int size) {
        CharacterProperties properties = CharacterPropertiesFactory.getCharacterProperties(font, color, size);
        properties.apply();
        content.append(c);
    }

    public void save(String filename) {
        System.out.println("Document saved as " + filename);
    }

    public static Document load(String filename) {
        System.out.println("Document loaded from " + filename);
        return new Document(); 
    }

    public void print() {
        System.out.println("Document content: " + content.toString());
    }
}