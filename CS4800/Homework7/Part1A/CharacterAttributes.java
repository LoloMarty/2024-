public class CharacterAttributes implements CharacterInterface {
    private final String font;
    private final String color;
    private final int size;

    public CharacterAttributes(String givenFont, String givenColor, int givenSize) {
        this.font = givenFont;
        this.color = givenColor;
        this.size = givenSize;
    }

    public void apply() {
        System.out.printf("Font: %s\nColor: %s\nSize: %d", this.font, this.color, this.size);
    }

    public String getFont() {
        return font;
    }

    public String getColor() {
        return color;
    }

    public int getSize() {
        return size;
    }
}
