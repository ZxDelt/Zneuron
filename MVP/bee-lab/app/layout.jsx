import "./globals.css";

export const metadata = {
  title: "Bee Lab",
  description: "Zneuron (Hub) vs LIF + R-STDP: two fly brains doing pick-and-place",
};

export const viewport = { width: "device-width", initialScale: 1, themeColor: "#07060f" };

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
