import org.junit.Test;

import static org.junit.Assert.*;

public class CharacterAttributesTest {

    @Test
    public void getFont() {
        String expectedFont = "Arial";
        String color = "Black";
        int size = 12;
        CharacterAttributes attributes = new CharacterAttributes(expectedFont, color, size);

        String actualFont = attributes.getFont();

        assertEquals(expectedFont, actualFont);
    }

    @Test
    public void getColor() {
        CharacterAttributes attributes = new CharacterAttributes("Arial", "Red", 12);

        String expectedColor = "Red";
        String actualColor = attributes.getColor();
        assertEquals(expectedColor, actualColor);
    }

    @Test
    public void getSize() {
        CharacterAttributes attributes = new CharacterAttributes("Arial", "Black", 12);

        int expectedSize = 12;
        int actualSize = attributes.getSize();
        assertEquals(expectedSize, actualSize);
    }
}