import "./globals.css";
import type {Metadata} from "next";
export const metadata:Metadata={title:"DeepCancer AI · Research workspace",description:"Explainable breast histopathology research"};
export default function Layout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>}
