package CS4800.HW4;

public class Pizza {
    private String chainName; 
    private String size; 
    private String[] toppings;

    private Pizza(PizzaBuilder builder)
    {
        this.size = builder.size;
        this.toppings = builder.toppings;
        this.chainName = builder.chainName;
    }

    public String getChainName()
    {
        return this.chainName;
    }

    public String getSize()
    {
        return this.size;
    }

    public String[] getToppings()
    {
        return this.toppings;
    }

    public void eat()
    {
        System.out.printf("\n\nChain Name: %s", this.getChainName());
        System.out.printf("\nPizza Size: %s", this.getSize());
        
        int toppingCounter = 1;
        for(String topping : this.getToppings())
        {
            System.out.printf("\nTopping #%d : %s", toppingCounter, topping);
            toppingCounter++;
        }
    }

    public static class PizzaBuilder{
        private String chainName; 
        private String size; 
        private String[] toppings; 

        public PizzaBuilder(String givenChainName)
        {
            this.chainName = givenChainName;
        }

        public PizzaBuilder setChainName(String givenChainName)
        {
            this.chainName = givenChainName;
            return this;
        }

        public PizzaBuilder setSize(String givenSize)
        {
            this.size = givenSize;
            return this;
        }

        public PizzaBuilder setToppings(String[] givenToppings)
        {
            this.toppings = givenToppings;
            return this;
        }

        public Pizza makePizza()
        {
            return new Pizza(this);
        }
    }

}
