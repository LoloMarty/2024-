public class Character {
    String heldCharacter;
    CharacterAttributes attributes;

    public Character(String givenCharacter, String givenFont, String givenColor, int givenSize) {
        this.heldCharacter = givenCharacter;

        CharacterAttributes fetchedAttributes = CharacterAttributesFactory.getCharacterProperties(givenFont,
                givenColor, givenSize);

        if (fetchedAttributes != null) {
            this.attributes = fetchedAttributes;
        } else {
            this.attributes = new CharacterAttributes(givenFont, givenColor, givenSize);
        }
    }

    public void printCharacaterAttributes() {
        this.attributes.apply();
    }

    public CharacterAttributes getAttributes() {
        return attributes;
    }

    public void setAttributes(CharacterAttributes attributes) {
        this.attributes = attributes;
    }

    public String getHeldCharacter() {
        return heldCharacter;
    }
}
