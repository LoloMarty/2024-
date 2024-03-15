package AbstractFactory;

public class Meal {
    private String carb;
    private String protein;
    private String fat;

    public Meal() {
        this.carb = "";
        this.protein = "";
        this.fat = "";
    }

    public String getCarb() {
        return carb;
    }

    public void setCarb(String carb) {
        this.carb = carb;
    }

    public String getProtein() {
        return protein;
    }

    public void setProtein(String protein) {
        this.protein = protein;
    }

    public String getFat() {
        return fat;
    }

    public void setFat(String fat) {
        this.fat = fat;
    }

}
