public class Main {
    public static void main(String[] args)
    {
        //IFood discount = new CustomerLoyalty(new FoodBurger( new FoodSandwich( new Food(2000, "Clam", new String[] {}))));

        IFood baseFood = new Food(2000, "Clam", new String[] {});
        IFood sandwich = new FoodSandwich(baseFood);
        IFood burger = new FoodBurger(sandwich);
        IFood discount = new CustomerLoyalty(burger);


        discount.calculateCost();
    }
}