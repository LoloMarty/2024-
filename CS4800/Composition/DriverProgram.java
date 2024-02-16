package CS4800.Composition;

public class DriverProgram {
    public static void main(String[] args)
    {
        Folder[] app_Folder_Subfolders = 
        {
            new Folder("config"),
            new Folder("controllers"),
            new Folder("library"),
            new Folder("migrations"),
            new Folder("models"),
            new Folder("views")
        };
    
        Folder app_Folder = new Folder(app_Folder_Subfolders, "app");
        
        File[] sourceFiles_Files = {
            new File(".htaccess"), 
            new File(".htrouter.php"), 
            new File("index.html")
        };
    
        Folder[] sourceFiles_Folder_Subfolders = {
            new Folder(".phalcon"),
            app_Folder,
            new Folder("cache"),
            new Folder("public")
        };
    
        Folder sourceFiles_Folder = new Folder(sourceFiles_Folder_Subfolders, sourceFiles_Files, "Source Files");
    
        Folder[] demo1_Folder_Subfolders = {
            sourceFiles_Folder, 
            new Folder("Include Path"),
            new Folder("Remote Files")
        };
    
        Folder demo1 = new Folder(demo1_Folder_Subfolders, "demo1");


        System.out.println();

        demo1.printEntireDirectory();

        System.out.println();
    }

}
