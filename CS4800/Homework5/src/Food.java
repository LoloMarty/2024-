import java.util.Hashtable;

public class Food implements IFood {
    private int basePrice;
    private String foodName;
    private String[] addedToppings;

    public Food(int basePrice, String foodName, String[] addedToppings)
    {
        this.basePrice = basePrice;
        this.foodName = foodName;
        this.addedToppings = addedToppings;
    }

    public Integer getToppingPrice(String topping)
    {
        return Toppings.getInstance().getToppingPrice(topping);
    }

    @Override
    public int calculateCost() {
        int additionalPrice = 0;
        for(String topping: this.addedToppings)
        {

            additionalPrice += this.getToppingPrice(topping);
        }

        System.out.printf("\nName: %s\nPrice (Euros): %d", this.foodName, this.basePrice+additionalPrice);
        return this.basePrice + additionalPrice;
    }
}
