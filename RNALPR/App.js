import * as React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";

import Home from "./components/home";
import Register from "./components/Registerpage";
import BookChallan from "./components/bookchallan";
import SendChallan from "./components/SendChallan";
import LoginScreen from "./components/login";
const Stack = createNativeStackNavigator();
export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator>
      <Stack.Screen name="LoginScreen" component={LoginScreen}  options={{
            headerShown: false,
          }}/>
        <Stack.Screen name="Home" component={Home}  options={{
            headerShown: false,
          }}/>
        <Stack.Screen name="Register" component={Register}  />
        <Stack.Screen name="BookChallan" component={BookChallan}  />
        <Stack.Screen name="SendChallan" component={SendChallan}  />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
