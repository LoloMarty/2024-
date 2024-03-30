import java.sql.Array;

public class FoodBurger extends FoodBase{

    public FoodBurger(IFood wrapped)
    {
        super(wrapped);
        this.basePrice = 4000;
        this.foodName = "Burger";
        this.addedToppings = new String[]{"Onions", "Gold Flakes", "Mushrooms"};
    }

    @Override
    public int calculateCost() {
        int carriedPrice = super.calculateCost();
        int additionalPrice = 0;

        for(String topping: this.addedToppings)
        {

            additionalPrice += this.getToppingPrice(topping);
        }

        System.out.printf("\nName: %s\nPrice (Euros): %d", this.foodName, this.basePrice+additionalPrice);
        return this.basePrice + additionalPrice + carriedPrice;
    }
}
