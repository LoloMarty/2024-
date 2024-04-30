import org.junit.Test;
import static org.junit.Assert.*;
import java.io.File;

public class CharStringTest {

    @Test
    public void save() {
        CharString charString = new CharString();

        charString.save("A", "Arial", "Black", 12);

        File file = new File("document.txt");
        assertTrue(file.exists());

        String expectedContent = "A";
        assertEquals(expectedContent, charString.buildString());
    }

    @Test
    public void buildString() {
        CharString charString = new CharString();
        charString.save("A", "Arial", "Black", 12);
        charString.save("B", "Times New Roman", "Red", 14);

        assertEquals("AB", charString.buildString());
    }
}