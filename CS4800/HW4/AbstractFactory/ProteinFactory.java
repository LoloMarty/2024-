package AbstractFactory;

public class ProteinFactory extends Nutrient {
    private String[] availableProteins = { "Fish", "Chicken", "Beef", "Tofu" };

    public String getAllowedFood(String[] foodExceptions) {
        String allowedFood = "";
        Boolean pass = true;

        for (String protein : this.availableProteins) {
            for (String food : foodExceptions) {
                if (protein.equalsIgnoreCase(food)) {
                    pass = false;
                }
            }

            if (pass == true) {
                allowedFood = protein;
            }

            pass = true;
        }

        return allowedFood;
    }
}
