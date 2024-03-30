import org.junit.Test;

import static org.junit.Assert.*;

public class ToppingsTest {

    @Test
    public void getToppingPrice() {
        int expected = 200;
        int actual = Toppings.getInstance().getToppingPrice("Bacon Bits");

        assertEquals(expected, actual);
    }
}