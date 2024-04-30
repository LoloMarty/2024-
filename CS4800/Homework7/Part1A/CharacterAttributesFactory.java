import java.util.HashMap;
import java.util.Map;

public class CharacterAttributesFactory {
    private static Map<String, CharacterAttributes> propertiesMap = new HashMap<>();

    public static CharacterAttributes getCharacterProperties(String font, String color, int size) {
        String key = font + color + size;
        if (!propertiesMap.containsKey(key)) {
            propertiesMap.put(key, new CharacterAttributes(font, color, size));
        }
        return propertiesMap.get(key);
    }

}
