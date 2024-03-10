package CS4800.HW4;

public class PizzaFactory {
    public static Pizza makePizza(String pizzaSize, String[] toppings)
    {
        Pizza returnedPizza = null;

        if (pizzaSize.equals("Small"))
        {
            returnedPizza = new SmallPizza(toppings);
        }else if (pizzaSize.equals("Medium"))
        {
            returnedPizza = new MedPizza(toppings);
        }else if (pizzaSize.equals("Large"))
        {
            returnedPizza = new LargePizza(toppings);
        }

        return returnedPizza;
    }
}
