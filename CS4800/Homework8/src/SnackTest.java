import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SnackTest {

    @Test
    public void testGetName() {
        Snack snack = new Snack("Chips", 2, 10);
        assertEquals("Chips", snack.getName());
    }

    @Test
    public void testGetPrice() {
        Snack snack = new Snack("Chips", 2, 10);
        assertEquals(2, snack.getPrice());
    }

    @Test
    public void testGetQuantity() {
        Snack snack = new Snack("Chips", 2, 10);
        assertEquals(10, snack.getQuantity());
    }

    @Test
    public void testSnackDispensed() {
        Snack snack = new Snack("Chips", 2, 10);
        snack.snackDispensed();
        assertEquals(9, snack.getQuantity());
    }

}