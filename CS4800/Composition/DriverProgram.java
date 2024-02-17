package CS4800.Composition;

public class DriverProgram {
    public static void main(String[] args) {

        Folder app_Folder = new Folder();
        app_Folder.setName("app");

        app_Folder.addFolder("config");
        app_Folder.addFolder("controllers");
        app_Folder.addFolder("library");
        app_Folder.addFolder("migrations");
        app_Folder.addFolder("models");
        app_Folder.addFolder("views");

        Folder sourceFiles_Folder = new Folder();
        sourceFiles_Folder.setName("Source Files");
        sourceFiles_Folder.addFile(".htaccess");
        sourceFiles_Folder.addFile(".htrouter.php");
        sourceFiles_Folder.addFile("index.html");

        sourceFiles_Folder.addFolder(".phalcon");
        sourceFiles_Folder.addFolder(app_Folder);
        sourceFiles_Folder.addFolder("cache");
        sourceFiles_Folder.addFolder("public");

        Folder demo1_Folder = new Folder();
        demo1_Folder.setName("demo1");
        demo1_Folder.addFolder(sourceFiles_Folder);
        demo1_Folder.addFolder("Include Path");
        demo1_Folder.addFolder("Remote Files");

        System.out.println();

        demo1_Folder.printEntireDirectory();

        System.out.println();

    }

}
