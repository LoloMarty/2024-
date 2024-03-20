package AbstractFactory;

public class CarbsFactory extends Nutrient {
    private String[] availableCarbs = { "Cheese", "Bread", "Lentils", "Pistachio" };

    public String getAllowedFood(String[] foodExceptions) {
        String allowedFood = "";
        Boolean pass = true;

        for (String carb : this.availableCarbs) {
            for (String food : foodExceptions) {
                if (carb.equalsIgnoreCase(food)) {
                    pass = false;
                }
            }

            if (pass == true) {
                allowedFood = carb;
            }

            pass = true;
        }

        return allowedFood;
    }
}
