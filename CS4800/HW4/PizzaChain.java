package CS4800.HW4;
import java.util.ArrayList;

public class PizzaChain {
    private String storeName; 
    private PizzaFactory pizzaFactory = new PizzaFactory();
    private ArrayList<Pizza> pizzaBuffer = new ArrayList<Pizza>();

    public PizzaChain(String name)
    {
        this.setStoreName(name);
    }

    public String getStoreName() {
        return storeName;
    }

    private void setStoreName(String storeName) {
        this.storeName = storeName;
    }

    public Pizza getPizza(String pizzaSize, String[] toppings)
    {
        Pizza newPizza = pizzaFactory.makePizza(pizzaSize, toppings);

        this.pizzaBuffer.add(newPizza);

        return newPizza;
    }
    
    public void eat(){
        for (Pizza pizza : pizzaBuffer)
        {
            System.out.printf("\n\nPizza Chain: \"%s\"", this.getStoreName());
            pizza.eat();
        }
    }
}
