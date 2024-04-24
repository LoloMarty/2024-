class ConcreteCharacterProperties implements CharacterProperties {
    private String font;
    private String color;
    private int size;

    public ConcreteCharacterProperties(String font, String color, int size) {
        this.font = font;
        this.color = color;
        this.size = size;
    }

    @Override
    public void apply() {
        System.out.println("Applying properties - Font: " + font + ", Color: " + color + ", Size: " + size);
    }
}
