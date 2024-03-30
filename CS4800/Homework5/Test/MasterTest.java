
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({
        CustomerLoyaltyTest.class,
        FoodBurgerTest.class,
        FoodSandwichTest.class,
        FoodTest.class,
        ToppingsTest.class,
})

public class MasterTest {
}
