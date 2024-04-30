import org.junit.Test;

import static org.junit.Assert.*;

public class CharacterTest {

    @Test
    public void setAttributes() {
        String initialFont = "Arial";
        String initialColor = "Red";
        int initialSize = 12;
        CharacterAttributes initialAttributes = new CharacterAttributes(initialFont, initialColor, initialSize);
        Character character = new Character("A", initialFont, initialColor, initialSize);

        String newFont = "Calibri";
        String newColor = "Blue";
        int newSize = 14;
        CharacterAttributes newAttributes = new CharacterAttributes(newFont, newColor, newSize);

        character.setAttributes(newAttributes);

        assertEquals(newFont, character.getAttributes().getFont());
        assertEquals(newColor, character.getAttributes().getColor());
        assertEquals(newSize, character.getAttributes().getSize());
    }
}