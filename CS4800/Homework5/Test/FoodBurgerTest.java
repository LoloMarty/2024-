import static org.junit.Assert.*;

public class FoodBurgerTest {

    @org.junit.Test
    public void calculateCost() {
        FoodBurger testCustomer = new FoodBurger(new Food(100, "TestFood", new String[]{}));

        int expected = 54250;
        int actual = testCustomer.calculateCost();

        assertEquals(expected, actual);
    }
}