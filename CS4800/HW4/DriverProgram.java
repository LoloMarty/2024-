package CS4800.HW4;

public class DriverProgram {
    public static void main(String[] args)
    {
        String[] threeToppings = {"Pepperoni", "Sausage", "Mushrooms"};
        String[] sixToppings = {"Pepperoni", "Sausage", "Mushrooms", "Bacon", "Onions", "Extra Cheese"};
        String[] nineToppings = {"Pepperoni", "Sausage", "Mushrooms", "Bacon", "Onions", "Extra Cheese", "Peppers", "Chicken", "Olives"};

        PizzaChain pizzaHutt = new PizzaChain("Pizza Hut");
        pizzaHutt.getPizza("Large", threeToppings);
        pizzaHutt.getPizza("Medium", sixToppings);
        pizzaHutt.getPizza("Small", nineToppings);


        pizzaHutt.eat();
    }
}
