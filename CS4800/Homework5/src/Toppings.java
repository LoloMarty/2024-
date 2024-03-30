import java.util.Hashtable;

class Toppings {
    private static Toppings toppingsInstance;
    private static Hashtable<String, Integer> allPossibleToppings;

    private Toppings()
    {
        allPossibleToppings = new Hashtable<>();
        allPossibleToppings.put("Onions", 120);
        allPossibleToppings.put("Tomatoes", 50);
        allPossibleToppings.put("Gold Flakes", 50000);
        allPossibleToppings.put("Bacon Bits", 200);
        allPossibleToppings.put("Ranch", 150);
        allPossibleToppings.put("Vegan Ranch", 149);
        allPossibleToppings.put("Fake Ranch", 151);
        allPossibleToppings.put("Mushrooms", 30);
    }

    public static Toppings getInstance()
    {
        if(toppingsInstance == null)
        {
            toppingsInstance = new Toppings();
        }

        return toppingsInstance;
    }

    public Integer getToppingPrice(String topping)
    {
        return allPossibleToppings.get(topping);
    }
}
