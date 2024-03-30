public class FoodSandwich extends FoodBase{
    public FoodSandwich(IFood wrapped)
    {
        super(wrapped);
        this.basePrice = 5000;
        this.foodName = "Sandwich";
        this.addedToppings = new String[]{"Ranch", "Vegan Ranch", "Bacon Bits"};
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
