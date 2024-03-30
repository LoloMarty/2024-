import org.junit.Test;

import static org.junit.Assert.*;

public class CustomerLoyaltyTest {

    @Test
    public void calculateCost() {
        CustomerLoyalty testCustomer = new CustomerLoyalty(new Food(100, "TestFood", new String[]{}));

        int expected = 9;
        int actual = testCustomer.calculateCost(3);

        assertEquals(expected, actual);
    }

    @Test
    public void getToppingsPrice()
    {
        CustomerLoyalty testCustomer = new CustomerLoyalty(new Food(100, "TestFood", new String[]{}));

        int expected = 151;
        int actual = testCustomer.getToppingPrice("Fake Ranch");

        assertEquals(expected, actual);
    }

}