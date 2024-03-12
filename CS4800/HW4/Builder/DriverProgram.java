package Builder;

public class DriverProgram {
        public static void main(String[] args) {
                String[] threeToppings = { "Pepperoni", "Sausage", "Mushrooms" };
                String[] sixToppings = { "Pepperoni", "Sausage", "Mushrooms", "Bacon", "Onions", "Extra Cheese" };
                String[] nineToppings = { "Pepperoni", "Sausage", "Mushrooms", "Bacon", "Onions", "Extra Cheese",
                                "Peppers",
                                "Chicken", "Olives" };

                /* HW 4, Part 1 */
                Pizza smlPizza = new Pizza.PizzaBuilder("Pizza Hut")
                                .setSize("Small")
                                .setToppings(threeToppings)
                                .makePizza();
                Pizza medPizza = new Pizza.PizzaBuilder("Pizza Hut")
                                .setSize("Medium")
                                .setToppings(sixToppings)
                                .makePizza();
                Pizza lrgPizza = new Pizza.PizzaBuilder("Pizza Hut")
                                .setSize("Large")
                                .setToppings(nineToppings)
                                .makePizza();

                smlPizza.eat();
                medPizza.eat();
                lrgPizza.eat();

                /* HW 4, Part 2 */
                Pizza pizzaHut_Large = new Pizza.PizzaBuilder("Pizza Hut")
                                .setSize("Large")
                                .setToppings(threeToppings)
                                .makePizza();
                Pizza pizzaHut_Small = new Pizza.PizzaBuilder("Pizza Hut")
                                .setSize("Small")
                                .setToppings(new String[] { "Tomato and Basil", "Beef" })
                                .makePizza();

                Pizza littleCaesars_Medium = new Pizza.PizzaBuilder("Little Caesars")
                                .setSize("Medium")
                                .setToppings(new String[] { "Tomato and Basil",
                                                "Spinach",
                                                "Ham",
                                                "Pesto",
                                                "Spicy Pork",
                                                "Olives",
                                                "Extra Cheese",
                                                "Onions" })
                                .makePizza();
                Pizza littleCaesars_Small = new Pizza.PizzaBuilder("Little Caesars")
                                .setSize("Small")
                                .setToppings(sixToppings)
                                .makePizza();

                Pizza dominos_Small = new Pizza.PizzaBuilder("Dominos")
                                .setSize("Small")
                                .setToppings(new String[] { "Spinach" })
                                .makePizza();

                Pizza dominos_Large = new Pizza.PizzaBuilder("Dominos")
                                .setSize("Large")
                                .setToppings(threeToppings)
                                .makePizza();

                pizzaHut_Large.eat();
                pizzaHut_Small.eat();

                littleCaesars_Medium.eat();
                littleCaesars_Small.eat();

                dominos_Large.eat();
                dominos_Small.eat();

        }
}
