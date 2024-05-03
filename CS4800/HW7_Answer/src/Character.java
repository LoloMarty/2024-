public class Character {
    private char character;
    private CharacterProperties characterProperty;

    public Character(char character, CharacterProperties characterProperty) {
        this.character = character;
        this.characterProperty = characterProperty;
    }
    public void setCharacter(char character) {
        this.character = character;
    }

    public void setCharacterProperty(CharacterProperties characterProperty) {
        this.characterProperty = characterProperty;
    }

    public char getCharacter() {
        return character;
    }

    public String getFont() {
        return characterProperty.getFont();
    }

    public String getColor() {
        return characterProperty.getColor();
    }

    public int getSize() {
        return characterProperty.getSize();
    }
}
