import org.junit.Test;

import static org.junit.Assert.*;

public class FoodTest {

    @Test
    public void getToppingPrice() {
        Food testFood = new Food(100, "TestFood", new String[]{});

        int expected = 120;
        int actual = testFood.getToppingPrice("Onions");

        assertEquals(expected, actual);
    }

    @Test
    public void calculateCost() {
        Food testFood = new Food(100, "TestFood", new String[]{"Onions"});

        int expected = 220;
        int actual = testFood.calculateCost();

        assertEquals(expected, actual);
    }
}