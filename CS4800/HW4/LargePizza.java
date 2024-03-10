package CS4800.HW4;

public class LargePizza extends Pizza{
    String[] toppings;

    public LargePizza (String[] toppings){
        this.setToppings(toppings);
    }

    public String[] getToppings() {
        return toppings;
    }

    public void setToppings(String[] toppings) {
        this.toppings = toppings;
    }

    public void eat(){
        int toppingCounter = 1;

        System.out.printf("\nPizza Size: Large");
        for(String topping : toppings)
        {
            System.out.printf("\nTopping %d: %s", toppingCounter, topping);
            toppingCounter++;
        }
    }
}