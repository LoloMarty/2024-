import org.junit.Test;

import static org.junit.Assert.*;

public class FoodSandwichTest {

    @Test
    public void calculateCost() {
        FoodSandwich testCustomer = new FoodSandwich(new Food(100, "TestFood", new String[]{}));

        int expected = 5599;
        int actual = testCustomer.calculateCost();

        assertEquals(expected, actual);
    }
}