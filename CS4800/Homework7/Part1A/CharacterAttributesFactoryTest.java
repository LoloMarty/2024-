import org.junit.Test;

import static org.junit.Assert.*;

public class CharacterAttributesFactoryTest {

    @Test
    public void getCharacterProperties() {
        String font = "Arial";
        String color = "black";
        int size = 12;

        CharacterAttributes attributes1 = CharacterAttributesFactory.getCharacterProperties(font, color, size);
        CharacterAttributes attributes2 = CharacterAttributesFactory.getCharacterProperties(font, color, size);

        assertNotNull(attributes1);
        assertNotNull(attributes2);
        assertEquals(attributes1, attributes2);
    }
}